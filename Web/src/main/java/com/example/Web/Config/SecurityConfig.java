package com.example.Web.Config;

import com.example.Web.Domain.UserRole;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;

@Configuration
@EnableWebSecurity
public class SecurityConfig extends WebSecurityConfigurerAdapter {

    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http.csrf().disable()
                .authorizeRequests() // 인증, 인가가 필요한 URL 지정정
                .antMatchers("/security-login/info").authenticated()
                .antMatchers("/security-login/admin/**").hasAnyAuthority(UserRole.ADMIN.name())
                .anyRequest().permitAll()
                .and()
                .formLogin()
                .usernameParameter("loginId")
                .passwordParameter("password")
                .loginPage("/security-login/login")
                .defaultSuccessUrl("/security-login")
                .failureUrl("/security-login/login")
                .and()
                .logout()
                .logoutUrl("/security-login/logout")
                .invalidateHttpSession(true).deleteCookies("JSESSIONID");
    }
}
