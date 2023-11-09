package com.example.Web.Domain;

import lombok.*;

import javax.persistence.*;

@Entity
@Table(name = "member")
@ToString
@Builder
@Getter @Setter
@AllArgsConstructor
@NoArgsConstructor
public class User {
    @Id @GeneratedValue
    private Long id;

    private String name;
    private String loginId;
    private String password;
    private String deviceId;

    private UserRole role;
}